"""Модели данных SQLAlchemy 2.0 для динамической онтологии медицинских протоколов.

Архитектура состоит из 3-х слоев:
1. Слой реестра схемы (DocumentProfile, FeatureNode, FeatureAlias) — развивающееся дерево признаков.
2. Слой фактов и происхождения (Document, DocumentValue) — разреженная матрица фактов.
3. Слой кэша готового JSON (DocumentJsonCache) — собранный валидный документ по схеме.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from src.config import settings


class Base(DeclarativeBase):
    """Базовый класс для моделей SQLAlchemy."""
    pass


# ============================================================================
# 1. СЛОЙ РЕЕСТРА СХЕМЫ И ПРИЗНАКОВ (SCHEMA & FEATURE REGISTRY)
# ============================================================================

class DocumentProfile(Base):
    """Справочник типов медицинских документов/протоколов (например, УЗИ ОБП, ЭхоКГ)."""

    __tablename__ = "document_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Связи
    features: Mapped[List["FeatureNode"]] = relationship(
        "FeatureNode",
        back_populates="profile",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="profile",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<DocumentProfile(id={self.id}, code='{self.code}', name='{self.name}')>"


class FeatureNode(Base):
    """Узел дерева онтологии признаков (органы, структуры, измерения).

    Организует древовидную вложенность JSON (например, 'liver' -> 'gallbladder' -> 'wall')
    через parent_id и full_path.
    """

    __tablename__ = "feature_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("feature_nodes.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    key_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    full_path: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data_type: Mapped[str] = mapped_column(
        String(32),
        default="string",
        nullable=False,
    )  # integer, float, string, dimension_2d, array, object
    unit: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_leaf: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        default="approved",
        nullable=False,
    )  # approved, candidate (для автоматически добавленных узлов)
    meta_data: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        server_default="{}",
        default=dict,
        nullable=False,
    )

    # Иерархические связи дерева признаков
    profile: Mapped["DocumentProfile"] = relationship("DocumentProfile", back_populates="features")
    parent: Mapped[Optional["FeatureNode"]] = relationship(
        "FeatureNode",
        remote_side="FeatureNode.id",
        back_populates="children",
        lazy="selectin",
    )
    children: Mapped[List["FeatureNode"]] = relationship(
        "FeatureNode",
        back_populates="parent",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    aliases: Mapped[List["FeatureAlias"]] = relationship(
        "FeatureAlias",
        back_populates="feature",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    values: Mapped[List["DocumentValue"]] = relationship(
        "DocumentValue",
        back_populates="feature",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("profile_id", "full_path", name="uq_profile_feature_path"),
    )

    def __repr__(self) -> str:
        return f"<FeatureNode(id={self.id}, path='{self.full_path}', type='{self.data_type}')>"


class FeatureAlias(Base):
    """Синонимы и варианты врачебных формулировок для конкретного узла схемы признаков."""

    __tablename__ = "feature_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("feature_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alias_text: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=True)

    feature: Mapped["FeatureNode"] = relationship("FeatureNode", back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("feature_id", "alias_text", name="uq_feature_alias"),
    )

    def __repr__(self) -> str:
        return f"<FeatureAlias(id={self.id}, feature_id={self.feature_id}, text='{self.alias_text}')>"


# ============================================================================
# 2. СЛОЙ ДОКУМЕНТОВ И ФАКТОВ (PROVENANCE & FACTS)
# ============================================================================

class Document(Base):
    """Исходный медицинский документ (протокол исследования, анкета)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document_profiles.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    consultation_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    administrative: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        server_default="{}",
        default=dict,
        nullable=False,
    )  # institution, doctor, age, date_of_birth, address, specialization, etc.
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        nullable=False,
    )

    profile: Mapped["DocumentProfile"] = relationship("DocumentProfile", back_populates="documents")
    values: Mapped[List["DocumentValue"]] = relationship(
        "DocumentValue",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    json_cache: Mapped[Optional["DocumentJsonCache"]] = relationship(
        "DocumentJsonCache",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, patient='{self.patient_id}', profile_id={self.profile_id})>"


class DocumentValue(Base):
    """Атомарные извлеченные факты протокола (разреженная матрица признаков).

    Хранит значение конкретного признака для конкретного документа с типизацией.
    Отсутствие строки для (document_id, feature_id) при формировании CSV становится NaN.
    """

    __tablename__ = "document_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("feature_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    value_numeric: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    value_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    value_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    raw_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    meta_data: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        server_default="{}",
        default=dict,
        nullable=False,
    )

    document: Mapped["Document"] = relationship("Document", back_populates="values")
    feature: Mapped["FeatureNode"] = relationship("FeatureNode", back_populates="values")

    __table_args__ = (
        UniqueConstraint("document_id", "feature_id", name="uq_document_feature_value"),
    )

    def __repr__(self) -> str:
        return f"<DocumentValue(id={self.id}, doc_id={self.document_id}, feature_id={self.feature_id})>"


# ============================================================================
# 3. СЛОЙ КЭША ГОТОВОГО ДОКУМЕНТА (TARGET JSON CACHE)
# ============================================================================

class DocumentJsonCache(Base):
    """Кэш итогового собранного JSON-документа (дерево признаков по целевой схеме)."""

    __tablename__ = "document_json_cache"

    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    compiled_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship("Document", back_populates="json_cache")

    def __repr__(self) -> str:
        return f"<DocumentJsonCache(document_id={self.document_id})>"


# ============================================================================
# 4. LEGACY МОДЕЛИ (для обратной совместимости на этапе рефакторинга)
# ============================================================================

class Parameter(Base):
    """[Legacy] Таблица канонических параметров онтологии."""

    __tablename__ = "parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=True)
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("parameters.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    meta_data: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        server_default="{}",
        default=dict,
        nullable=False,
    )
    synonyms: Mapped[List["ParameterSynonym"]] = relationship(
        "ParameterSynonym",
        back_populates="parameter",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    records: Mapped[List["PatientRecord"]] = relationship(
        "PatientRecord",
        back_populates="parameter",
        lazy="selectin",
    )
    parent: Mapped[Optional["Parameter"]] = relationship(
        "Parameter",
        remote_side="Parameter.id",
        back_populates="children",
        lazy="selectin",
    )
    children: Mapped[List["Parameter"]] = relationship(
        "Parameter",
        back_populates="parent",
        lazy="selectin",
    )


class ParameterSynonym(Base):
    """[Legacy] Таблица синонимов параметров."""

    __tablename__ = "parameter_synonyms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parameter_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("parameters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    synonym_text: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIM), nullable=True)
    parameter: Mapped["Parameter"] = relationship(
        "Parameter",
        back_populates="synonyms",
    )
    __table_args__ = (
        UniqueConstraint("parameter_id", "synonym_text", name="uq_param_synonym"),
    )


class PatientRecord(Base):
    """[Legacy] Таблица записей пациентов."""

    __tablename__ = "patient_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    parameter_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("parameters.id"),
        nullable=True,
        index=True,
    )
    raw_key: Mapped[str] = mapped_column(Text, nullable=False)
    raw_value: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        nullable=False,
    )
    meta_data: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        server_default="{}",
        default=dict,
        nullable=False,
    )
    parameter: Mapped[Optional["Parameter"]] = relationship(
        "Parameter",
        back_populates="records",
    )
